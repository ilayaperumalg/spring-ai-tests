package com.example.demo;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.testcontainers.chromadb.ChromaDBContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import org.springframework.ai.chroma.vectorstore.ChromaApi;
import org.springframework.ai.chroma.vectorstore.ChromaVectorStore;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@Testcontainers
class DemoApplicationTests {

	@Container
	static ChromaDBContainer chromaContainer = new ChromaDBContainer("ghcr.io/chroma-core/chroma:0.5.20");

	@Autowired
	VectorStore vectorStore;

	@Test
	void contextLoads() {
		writeToChrome("Spring AI rocks!");
		writeToChrome("Summer coding!");
		writeToChrome("Winter break!");

		List<Document> results = vectorStore
				.similaritySearch(SearchRequest.builder().query("Spring").topK(1).build());

		assertThat(results).hasSize(1);
	}

	void writeToChrome(String text) {
		Resource resource = new ByteArrayResource(text.getBytes());
		TextReader reader = new TextReader(resource);
		TokenTextSplitter splitter = new TokenTextSplitter();
		List<Document> split = splitter.split(reader.read());
		this.vectorStore.write(split);
	}

	@SpringBootConfiguration
	public static class TestApplication {

		@Bean
		public RestClient.Builder builder() {
			return RestClient.builder().requestFactory(new SimpleClientHttpRequestFactory());
		}

		@Bean
		public ChromaApi chromaApi(RestClient.Builder builder) {
			return new ChromaApi(chromaContainer.getEndpoint(), builder);
		}

		@Bean
		public VectorStore chromaVectorStore(EmbeddingModel embeddingModel, ChromaApi chromaApi) {
			return ChromaVectorStore.builder(chromaApi, embeddingModel)
					.collectionName("TestCollection")
					.initializeSchema(true)
					.build();
		}

		@Bean
		public EmbeddingModel embeddingModel() {
			return new OpenAiEmbeddingModel(new OpenAiApi(System.getenv("OPENAI_API_KEY")));
		}

	}

}
